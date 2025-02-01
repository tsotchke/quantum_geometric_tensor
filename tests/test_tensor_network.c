#include "../include/quantum_geometric/core/quantum_geometric_tensor_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* Test configuration */
#define TEST_DIMENSION 32
#define TEST_NUM_SPINS 16
#define TEST_BOND_DIM 8
#define TEST_TOLERANCE 1e-6
#define PERF_ITERATIONS 1000

/* Test macros */
#define ASSERT_NEAR(x, y, tol) assert(fabs((x) - (y)) < (tol))
#define ASSERT_SUCCESS(x) assert((x) == QGT_SUCCESS)
#define ASSERT_PHYSICSML_SUCCESS(x) assert((x) == PHYSICSML_SUCCESS)

/* Test utilities */
static void init_random_tensor(quantum_geometric_tensor* tensor) {
    for (size_t i = 0; i < tensor->num_spins; i++) {
        double real = (double)rand() / RAND_MAX;
        double imag = (double)rand() / RAND_MAX;
        double norm = sqrt(real * real + imag * imag);
        tensor->spin_system.spin_states[i] = (real + I * imag) / norm;
    }
}

/* Test tensor conversion */
static void test_tensor_conversion() {
    printf("Testing tensor conversion...\n");
    
    // Create quantum geometric tensor
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_HUGE_PAGES
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);
    
    // Convert to PhysicsML tensor
    PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
    assert(pml != NULL);
    
    // Convert back to quantum geometric tensor
    quantum_geometric_tensor* qgt2 = physicsml_to_qgt_tensor(pml);
    assert(qgt2 != NULL);
    
    // Verify consistency
    assert(verify_tensor_consistency(qgt, pml, TEST_TOLERANCE));
    assert(verify_tensor_consistency(qgt2, pml, TEST_TOLERANCE));
    
    // Cleanup
    free_quantum_tensor(qgt);
    free_quantum_tensor(qgt2);
    physicsml_tensor_destroy(pml);
    
    printf("✓ Tensor conversion tests passed\n");
}

/* Test tensor network creation */
static void test_network_creation() {
    printf("Testing network creation...\n");
    
    // Create quantum geometric tensor
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_HUGE_PAGES
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);
    
    // Create tensor network
    TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
    assert(ttn != NULL);
    
    // Extract properties
    quantum_geometric_tensor* extracted = extract_geometric_properties(ttn);
    assert(extracted != NULL);
    
    // Verify properties preserved
    PhysicsMLTensor* pml = physicsml_contract_network(ttn);
    assert(pml != NULL);
    assert(verify_tensor_consistency(qgt, pml, TEST_TOLERANCE));
    
    // Cleanup
    free_quantum_tensor(qgt);
    free_quantum_tensor(extracted);
    physicsml_tensor_destroy(pml);
    physicsml_ttn_destroy(ttn);
    
    printf("✓ Network creation tests passed\n");
}

/* Test physical constraints */
static void test_physical_constraints() {
    printf("Testing physical constraints...\n");
    
    // Create quantum geometric tensor
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_HUGE_PAGES
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);
    
    // Create constraints
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
    
    // Apply constraints
    ASSERT_SUCCESS(apply_physical_constraints(qgt, &constraints));
    
    // Verify constraints satisfied
    PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
    assert(pml != NULL);
    
    // Check energy constraint
    double energy = 0.0;
    complex double* data = (complex double*)pml->data;
    for (size_t i = 0; i < pml->size; i++) {
        energy += cabs(data[i]) * cabs(data[i]);
    }
    ASSERT_NEAR(energy, constraints.energy_threshold, TEST_TOLERANCE);
    
    // Check symmetry constraint
    for (size_t i = 0; i < TEST_DIMENSION; i++) {
        for (size_t j = i + 1; j < TEST_DIMENSION; j++) {
            size_t ij = i * TEST_DIMENSION + j;
            size_t ji = j * TEST_DIMENSION + i;
            complex double diff = data[ij] - conj(data[ji]);
            assert(cabs(diff) < constraints.symmetry_tolerance);
        }
    }
    
    // Cleanup
    free_quantum_tensor(qgt);
    physicsml_tensor_destroy(pml);
    
    printf("✓ Physical constraints tests passed\n");
}

/* Test geometric constraints */
static void test_geometric_constraints() {
    printf("Testing geometric constraints...\n");
    
    // Create quantum geometric tensor
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_HUGE_PAGES
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);
    
    // Create tensor network
    TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
    assert(ttn != NULL);
    
    // Apply geometric constraints
    ASSERT_PHYSICSML_SUCCESS(apply_geometric_constraints(ttn, qgt));
    
    // Verify constraints preserved
    quantum_geometric_tensor* extracted = extract_geometric_properties(ttn);
    assert(extracted != NULL);
    
    // Check metric tensor preserved
    for (size_t i = 0; i < qgt->dimension * qgt->dimension; i++) {
        ASSERT_NEAR(qgt->geometry.metric_tensor[i],
                   extracted->geometry.metric_tensor[i],
                   TEST_TOLERANCE);
    }
    
    // Cleanup
    free_quantum_tensor(qgt);
    free_quantum_tensor(extracted);
    physicsml_ttn_destroy(ttn);
    
    printf("✓ Geometric constraints tests passed\n");
}

/* Performance benchmarks */
static void benchmark_tensor_operations() {
    printf("Running performance benchmarks...\n");
    
    // Create test tensors
    quantum_geometric_tensor* qgt = create_quantum_tensor(
        TEST_DIMENSION, TEST_NUM_SPINS, QGT_MEM_HUGE_PAGES
    );
    assert(qgt != NULL);
    init_random_tensor(qgt);
    
    // Benchmark tensor conversion
    clock_t start = clock();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
        assert(pml != NULL);
        physicsml_tensor_destroy(pml);
    }
    clock_t conversion_time = clock() - start;
    
    // Benchmark network creation
    start = clock();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        TreeTensorNetwork* ttn = create_geometric_network(qgt, TEST_BOND_DIM);
        assert(ttn != NULL);
        physicsml_ttn_destroy(ttn);
    }
    clock_t network_time = clock() - start;
    
    // Benchmark constraint application
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
    
    // Report results
    printf("Performance results:\n");
    printf("- Tensor conversion: %f ms/op\n", 
           1000.0 * conversion_time / CLOCKS_PER_SEC / PERF_ITERATIONS);
    printf("- Network creation: %f ms/op\n",
           1000.0 * network_time / CLOCKS_PER_SEC / PERF_ITERATIONS);
    printf("- Constraint application: %f ms/op\n",
           1000.0 * constraint_time / CLOCKS_PER_SEC / PERF_ITERATIONS);
    
    // Cleanup
    free_quantum_tensor(qgt);
    
    printf("✓ Performance benchmarks completed\n");
}

/* Main test runner */
int main() {
    printf("\nQuantum Geometric Tensor Network Test Suite\n");
    printf("=========================================\n\n");
    
    /* Initialize random seed */
    srand(time(NULL));
    
    /* Run tests */
    test_tensor_conversion();
    test_network_creation();
    test_physical_constraints();
    test_geometric_constraints();
    benchmark_tensor_operations();
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
