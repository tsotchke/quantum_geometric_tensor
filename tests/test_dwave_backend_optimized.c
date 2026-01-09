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
#include <string.h>

// Test helper functions
static quantum_problem* create_test_problem() {
    quantum_problem* problem = malloc(sizeof(quantum_problem));
    problem->num_qubits = 4;
    problem->num_terms = 0;
    problem->capacity = 100;
    problem->terms = calloc(problem->capacity, sizeof(quantum_term));
    problem->energy_offset = 0.0;
    return problem;
}

static void cleanup_test_problem(quantum_problem* problem) {
    if (problem) {
        free(problem->terms);
        free(problem);
    }
}

static void add_term_to_problem(quantum_problem* problem, size_t* qubits, size_t num_qubits, double coefficient) {
    if (problem->num_terms < problem->capacity) {
        quantum_term* term = &problem->terms[problem->num_terms];
        term->num_qubits = num_qubits;
        for (size_t i = 0; i < num_qubits && i < MAX_TERM_QUBITS; i++) {
            term->qubits[i] = qubits[i];
        }
        term->coefficient = coefficient;
        problem->num_terms++;
    }
}

static void test_initialization() {
    printf("Testing D-Wave backend initialization...\n");

    // Setup sampling parameters
    DWaveSamplingParams sampling_params = {
        .num_reads = 1000,
        .annealing_time = 20,
        .chain_strength = 2.0,
        .programming_thermalization = 100,
        .auto_scale = true,
        .reduce_intersample_correlation = false,
        .readout_thermalization = NULL,
        .custom_params = NULL
    };

    // Setup backend config
    DWaveBackendConfig config = {
        .type = DWAVE_BACKEND_SIMULATOR,  // Use simulator for testing
        .api_token = NULL,
        .solver_name = "DW_2000Q_6",
        .solver_type = DWAVE_SOLVER_DW2000Q,
        .problem_type = DWAVE_PROBLEM_ISING,
        .sampling_params = sampling_params,
        .custom_config = NULL
    };

    // Initialize backend
    DWaveConfig* dwave_config = init_dwave_backend(&config);
    assert(dwave_config != NULL && "Failed to initialize backend");

    // Cleanup
    cleanup_dwave_config(dwave_config);
    printf("Initialization test passed\n");
}

static void test_problem_creation() {
    printf("Testing D-Wave problem creation...\n");

    // Create problem using D-Wave API
    DWaveProblem* problem = create_dwave_problem(4, 6);
    assert(problem != NULL && "Failed to create D-Wave problem");
    assert(problem->num_variables == 4 && "Wrong number of variables");

    // Add linear terms (h)
    assert(add_linear_term(problem, 0, -1.0) && "Failed to add linear term");
    assert(add_linear_term(problem, 1, 0.5) && "Failed to add linear term");
    assert(add_linear_term(problem, 2, -0.5) && "Failed to add linear term");
    assert(add_linear_term(problem, 3, 1.0) && "Failed to add linear term");

    // Add quadratic terms (J)
    assert(add_quadratic_term(problem, 0, 1, 1.0) && "Failed to add quadratic term");
    assert(add_quadratic_term(problem, 0, 2, 1.0) && "Failed to add quadratic term");
    assert(add_quadratic_term(problem, 0, 3, 1.0) && "Failed to add quadratic term");
    assert(add_quadratic_term(problem, 1, 2, 1.0) && "Failed to add quadratic term");
    assert(add_quadratic_term(problem, 1, 3, 1.0) && "Failed to add quadratic term");
    assert(add_quadratic_term(problem, 2, 3, 1.0) && "Failed to add quadratic term");

    // Cleanup
    cleanup_dwave_problem(problem);
    printf("Problem creation test passed\n");
}

static void test_quantum_problem_structure() {
    printf("Testing quantum problem structure...\n");

    // Create test problem
    quantum_problem* problem = create_test_problem();
    assert(problem != NULL && "Failed to create test problem");
    assert(problem->num_qubits == 4 && "Wrong number of qubits");
    assert(problem->capacity == 100 && "Wrong capacity");

    // Add a 2-qubit term (interaction)
    size_t qubits_2[] = {0, 1};
    add_term_to_problem(problem, qubits_2, 2, 1.0);

    // Add a 3-qubit term
    size_t qubits_3[] = {0, 2, 3};
    add_term_to_problem(problem, qubits_3, 3, -0.5);

    // Add single qubit terms (local fields)
    size_t qubits_1a[] = {0};
    add_term_to_problem(problem, qubits_1a, 1, 2.0);

    size_t qubits_1b[] = {1};
    add_term_to_problem(problem, qubits_1b, 1, -1.5);

    assert(problem->num_terms == 4 && "Wrong number of terms");

    // Verify first term
    assert(problem->terms[0].num_qubits == 2 && "Wrong number of qubits in term 0");
    assert(problem->terms[0].qubits[0] == 0 && "Wrong qubit index in term 0");
    assert(problem->terms[0].qubits[1] == 1 && "Wrong qubit index in term 0");
    assert(fabs(problem->terms[0].coefficient - 1.0) < 1e-6 && "Wrong coefficient in term 0");

    // Cleanup
    cleanup_test_problem(problem);
    printf("Quantum problem structure test passed\n");
}

static void test_format_conversion() {
    printf("Testing QUBO to Ising conversion...\n");

    // Create QUBO problem
    DWaveProblem* qubo = create_dwave_problem(3, 3);
    assert(qubo != NULL && "Failed to create QUBO problem");

    // Add QUBO terms
    add_linear_term(qubo, 0, 1.0);
    add_linear_term(qubo, 1, 1.0);
    add_linear_term(qubo, 2, 1.0);
    add_quadratic_term(qubo, 0, 1, 2.0);
    add_quadratic_term(qubo, 1, 2, 2.0);

    // Convert to Ising
    DWaveProblem* ising = qubo_to_ising(qubo);
    assert(ising != NULL && "Failed to convert QUBO to Ising");
    assert(ising->num_variables == qubo->num_variables && "Variable count changed");

    // Convert back to QUBO
    DWaveProblem* qubo2 = ising_to_qubo(ising);
    assert(qubo2 != NULL && "Failed to convert Ising to QUBO");

    // Cleanup
    cleanup_dwave_problem(qubo);
    cleanup_dwave_problem(ising);
    cleanup_dwave_problem(qubo2);
    printf("Format conversion test passed\n");
}

static void test_job_submission() {
    printf("Testing job submission and result retrieval...\n");

    // Setup backend
    DWaveSamplingParams sampling_params = {
        .num_reads = 100,
        .annealing_time = 20,
        .chain_strength = 1.5,
        .programming_thermalization = 50,
        .auto_scale = true,
        .reduce_intersample_correlation = false,
        .readout_thermalization = NULL,
        .custom_params = NULL
    };

    DWaveBackendConfig config = {
        .type = DWAVE_BACKEND_SIMULATOR,
        .api_token = NULL,
        .solver_name = "simulator",
        .solver_type = DWAVE_SOLVER_NEAL,
        .problem_type = DWAVE_PROBLEM_ISING,
        .sampling_params = sampling_params,
        .custom_config = NULL
    };

    DWaveConfig* dwave_config = init_dwave_backend(&config);
    assert(dwave_config != NULL && "Failed to initialize backend");

    // Create simple problem
    DWaveProblem* problem = create_dwave_problem(3, 3);
    add_linear_term(problem, 0, -1.0);
    add_linear_term(problem, 1, -1.0);
    add_linear_term(problem, 2, -1.0);
    add_quadratic_term(problem, 0, 1, 2.0);
    add_quadratic_term(problem, 1, 2, 2.0);

    // Setup job config
    DWaveJobConfig job_config = {
        .problem = problem,
        .params = sampling_params,
        .use_embedding = false,
        .use_error_mitigation = false,
        .custom_options = NULL
    };
    memset(job_config.job_tags, 0, sizeof(job_config.job_tags));

    // Submit job
    char* job_id = submit_dwave_job(dwave_config, &job_config);
    assert(job_id != NULL && "Failed to submit job");

    // Check status
    DWaveJobStatus status = get_dwave_job_status(dwave_config, job_id);
    assert(status != DWAVE_STATUS_ERROR && "Job failed");

    // Get result
    DWaveJobResult* result = get_dwave_job_result(dwave_config, job_id);
    assert(result != NULL && "Failed to get job result");
    assert(result->num_samples > 0 && "No samples returned");

    // Verify result structure
    assert(result->samples != NULL && "Samples array is NULL");
    assert(result->energies != NULL && "Energies array is NULL");

    // Cleanup
    free(job_id);
    cleanup_dwave_result(result);
    cleanup_dwave_problem(problem);
    cleanup_dwave_config(dwave_config);
    printf("Job submission test passed\n");
}

int main() {
    printf("Running D-Wave backend tests\n");
    printf("============================\n\n");

    test_initialization();
    printf("\n");

    test_problem_creation();
    printf("\n");

    test_quantum_problem_structure();
    printf("\n");

    test_format_conversion();
    printf("\n");

    test_job_submission();
    printf("\n");

    printf("All D-Wave backend tests passed!\n");
    return 0;
}
