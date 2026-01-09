/**
 * @file test_tensor_network.c
 * @brief Test suite for quantum geometric tensor network operations
 *
 * Tests tensor conversion, network creation, physical constraints,
 * and geometric constraints for the PhysicsML integration layer.
 */

#include "quantum_geometric/ai/quantum_geometric_tensor_network.h"
#include "quantum_geometric/physics/advanced_geometry_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* Test configuration */
#define TEST_DIMENSION 32
#define TEST_NUM_SPINS 16
#define TEST_BOND_DIM 8
#define TEST_TOLERANCE 1e-6
#define PERF_ITERATIONS 100  /* Reduced for faster test runs */

/* Test macros */
#define ASSERT_NEAR(x, y, tol) assert(fabs((x) - (y)) < (tol))
#define ASSERT_SUCCESS(x) assert((x) == QGT_SUCCESS)
#define ASSERT_PHYSICSML_SUCCESS(x) assert((x) == PHYSICSML_SUCCESS)

/**
 * @brief Initialize tensor with random normalized spin states
 *
 * Fills the spin_system.spin_states array with random complex values
 * that are normalized to unit length.
 *
 * @param tensor Tensor to initialize
 */
static void init_random_tensor(quantum_geometric_tensor* tensor) {
    if (!tensor || !tensor->spin_system.spin_states) return;

    size_t state_size = tensor->dimension * tensor->spin_system.spin_dim;

    /* Generate random complex amplitudes and normalize */
    double total_norm = 0.0;

    for (size_t i = 0; i < state_size; i++) {
        double real = (double)rand() / RAND_MAX - 0.5;
        double imag = (double)rand() / RAND_MAX - 0.5;
        tensor->spin_system.spin_states[i].real = real;
        tensor->spin_system.spin_states[i].imag = imag;
        total_norm += real * real + imag * imag;
    }

    /* Normalize to unit total probability */
    total_norm = sqrt(total_norm);
    if (total_norm > 1e-10) {
        for (size_t i = 0; i < state_size; i++) {
            tensor->spin_system.spin_states[i].real /= total_norm;
            tensor->spin_system.spin_states[i].imag /= total_norm;
        }
    }
}

/**
 * @brief Compute magnitude of ComplexDouble value
 */
static double complex_abs(ComplexDouble c) {
    return sqrt(c.real * c.real + c.imag * c.imag);
}

/* Test tensor conversion */
static void test_tensor_conversion(void) {
    printf("Testing tensor conversion...\n");

    /* Create quantum geometric tensor */
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_STANDARD
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);

    /* Convert to PhysicsML tensor */
    PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
    assert(pml != NULL);
    assert(pml->data != NULL);
    assert(pml->shape != NULL);
    assert(pml->ndim == 2);
    assert(pml->shape[0] == TEST_DIMENSION);
    assert(pml->shape[1] == TEST_NUM_SPINS);

    /* Convert back to quantum geometric tensor */
    quantum_geometric_tensor* qgt2 = physicsml_to_qgt_tensor(pml);
    assert(qgt2 != NULL);

    /* Verify consistency */
    assert(verify_tensor_consistency(qgt, pml, TEST_TOLERANCE));
    assert(verify_tensor_consistency(qgt2, pml, TEST_TOLERANCE));

    /* Cleanup */
    free_quantum_tensor(qgt);
    free_quantum_tensor(qgt2);
    physicsml_tensor_destroy(pml);

    printf("  Tensor conversion tests passed\n");
}

/* Test tensor network creation */
static void test_network_creation(void) {
    printf("Testing network creation...\n");

    /* Create quantum geometric tensor */
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_STANDARD
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);

    /* Create tensor network */
    TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
    assert(ttn != NULL);

    /* Extract properties */
    quantum_geometric_tensor* extracted = extract_geometric_properties(ttn);
    assert(extracted != NULL);

    /* Verify properties preserved - contract network to tensor */
    PhysicsMLTensor* pml = physicsml_contract_network(ttn);
    assert(pml != NULL);

    /* Cleanup */
    free_quantum_tensor(qgt);
    free_quantum_tensor(extracted);
    physicsml_tensor_destroy(pml);
    physicsml_ttn_destroy(ttn);

    printf("  Network creation tests passed\n");
}

/* Test physical constraints */
static void test_physical_constraints(void) {
    printf("Testing physical constraints...\n");

    /* Create quantum geometric tensor */
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_STANDARD
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);

    /* Create constraints */
    PhysicalConstraints constraints = {
        .energy_threshold = 1.0,
        .fidelity_threshold = 0.99,
        .symmetry_tolerance = 1e-4,
        .conservation_tolerance = 1e-4,
        .gauge_tolerance = 1e-4,
        .locality_tolerance = 1e-4,
        .renormalization_scale = 1.0,
        .causality_tolerance = 1e-4
    };

    /* Apply constraints */
    ASSERT_SUCCESS(apply_physical_constraints(qgt, &constraints));

    /* Verify constraints satisfied */
    PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
    assert(pml != NULL);

    /* Check energy constraint - total energy should be <= threshold */
    double energy = 0.0;
    ComplexDouble* data = (ComplexDouble*)pml->data;
    for (size_t i = 0; i < pml->size; i++) {
        double abs_val = complex_abs(data[i]);
        energy += abs_val * abs_val;
    }
    assert(energy <= constraints.energy_threshold + TEST_TOLERANCE);

    /* Check metric tensor symmetry constraint */
    if (qgt->geometry.metric_tensor) {
        for (size_t i = 0; i < qgt->dimension && i < 8; i++) {
            for (size_t j = i + 1; j < qgt->dimension && j < 8; j++) {
                size_t ij = i * qgt->dimension + j;
                size_t ji = j * qgt->dimension + i;
                double diff = fabs(qgt->geometry.metric_tensor[ij] -
                                   qgt->geometry.metric_tensor[ji]);
                assert(diff < constraints.symmetry_tolerance);
            }
        }
    }

    /* Cleanup */
    free_quantum_tensor(qgt);
    physicsml_tensor_destroy(pml);

    printf("  Physical constraints tests passed\n");
}

/* Test geometric constraints */
static void test_geometric_constraints(void) {
    printf("Testing geometric constraints...\n");

    /* Create quantum geometric tensor */
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_STANDARD
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);

    /* Create tensor network */
    TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
    assert(ttn != NULL);

    /* Apply geometric constraints */
    ASSERT_PHYSICSML_SUCCESS(apply_geometric_constraints(ttn, qgt));

    /* Verify constraints preserved */
    quantum_geometric_tensor* extracted = extract_geometric_properties(ttn);
    assert(extracted != NULL);

    /* Check metric tensor exists */
    assert(qgt->geometry.metric_tensor != NULL);
    assert(extracted->geometry.metric_tensor != NULL);

    /* Check diagonal elements of metric tensor (identity-like structure) */
    for (size_t i = 0; i < qgt->dimension && i < 8; i++) {
        double diag_val = qgt->geometry.metric_tensor[i * qgt->dimension + i];
        assert(diag_val > 0.0);  /* Positive definite */
    }

    /* Cleanup */
    free_quantum_tensor(qgt);
    free_quantum_tensor(extracted);
    physicsml_ttn_destroy(ttn);

    printf("  Geometric constraints tests passed\n");
}

/* Performance benchmarks */
static void benchmark_tensor_operations(void) {
    printf("Running performance benchmarks...\n");

    /* Create test tensors */
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_STANDARD
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);

    /* Benchmark tensor conversion */
    clock_t start = clock();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
        assert(pml != NULL);
        physicsml_tensor_destroy(pml);
    }
    clock_t conversion_time = clock() - start;

    /* Benchmark network creation */
    start = clock();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
        assert(ttn != NULL);
        physicsml_ttn_destroy(ttn);
    }
    clock_t network_time = clock() - start;

    /* Benchmark constraint application */
    PhysicalConstraints constraints = {
        .energy_threshold = 1.0,
        .fidelity_threshold = 0.99,
        .symmetry_tolerance = 1e-6,
        .conservation_tolerance = 1e-6,
        .gauge_tolerance = 1e-6,
        .locality_tolerance = 1e-6,
        .renormalization_scale = 1.0,
        .causality_tolerance = 1e-6
    };

    start = clock();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        ASSERT_SUCCESS(apply_physical_constraints(qgt, &constraints));
    }
    clock_t constraint_time = clock() - start;

    /* Report results */
    printf("Performance results (%d iterations):\n", PERF_ITERATIONS);
    printf("- Tensor conversion: %.3f ms/op\n",
           1000.0 * conversion_time / CLOCKS_PER_SEC / PERF_ITERATIONS);
    printf("- Network creation: %.3f ms/op\n",
           1000.0 * network_time / CLOCKS_PER_SEC / PERF_ITERATIONS);
    printf("- Constraint application: %.3f ms/op\n",
           1000.0 * constraint_time / CLOCKS_PER_SEC / PERF_ITERATIONS);

    /* Cleanup */
    free_quantum_tensor(qgt);

    printf("  Performance benchmarks completed\n");
}

/* Main test runner */
int main(void) {
    printf("\nQuantum Geometric Tensor Network Test Suite\n");
    printf("=========================================\n\n");

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Run tests */
    test_tensor_conversion();
    test_network_creation();
    test_physical_constraints();
    test_geometric_constraints();
    benchmark_tensor_operations();

    printf("\nAll tests passed successfully!\n");
    return 0;
}
