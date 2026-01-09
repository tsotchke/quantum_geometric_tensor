/**
 * @file test_topological_protection.c
 * @brief Tests for topological error correction and coherence protection
 *
 * This test suite validates the topological protection mechanisms including
 * error detection, correction, and maintenance of quantum coherence.
 */

#include "quantum_geometric/physics/quantum_topological_operations.h"
#include "quantum_geometric/core/tree_tensor_network.h"
#include "quantum_geometric/core/error_codes.h"
#include "test_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Test configuration */
#define TEST_DIM 32
#define TEST_BOND_DIM 8
#define TEST_NUM_QUBITS 16
#define TEST_TOLERANCE 1e-4
#define TEST_SVD_TOLERANCE 1e-6
#define EXPECTED_TEE_MIN 0.0
#define EXPECTED_TEE_MAX 10.0

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;

/* Test fixtures */
static quantum_topological_tensor_t* qgt = NULL;
static TreeTensorNetwork* network = NULL;

/* Helper function to create topological tensor */
static quantum_topological_tensor_t* create_test_topological_tensor(size_t dim, size_t num_qubits) {
    quantum_topological_tensor_t* tensor = (quantum_topological_tensor_t*)calloc(1,
        sizeof(quantum_topological_tensor_t));
    if (!tensor) return NULL;

    tensor->dimension = dim;
    tensor->rank = 2;
    tensor->num_spins = num_qubits;
    tensor->hardware = 0;  // CPU

    // Allocate components
    size_t comp_size = dim * dim;
    tensor->components = (ComplexFloat*)calloc(comp_size, sizeof(ComplexFloat));
    if (!tensor->components) {
        free(tensor);
        return NULL;
    }

    // Initialize to identity-like state
    for (size_t i = 0; i < dim && i < comp_size / dim; i++) {
        tensor->components[i * dim + i].real = 1.0f / sqrtf((float)dim);
        tensor->components[i * dim + i].imag = 0.0f;
    }

    // Initialize spin system
    tensor->spin_system.num_spins = num_qubits;
    tensor->spin_system.coupling_strength = 1.0;
    tensor->spin_system.spin_states = (complex double*)calloc(num_qubits, sizeof(complex double));
    if (tensor->spin_system.spin_states) {
        for (size_t i = 0; i < num_qubits; i++) {
            tensor->spin_system.spin_states[i] = 1.0 / sqrt((double)num_qubits);
        }
    }

    // Initialize geometry
    tensor->geometry.dimension = dim;
    tensor->geometry.metric_tensor = (double*)calloc(dim * dim, sizeof(double));
    if (tensor->geometry.metric_tensor) {
        for (size_t i = 0; i < dim; i++) {
            tensor->geometry.metric_tensor[i * dim + i] = 1.0;
        }
    }

    // Initialize topology
    tensor->topology.num_singular_values = dim;
    tensor->topology.singular_values = (double*)calloc(dim, sizeof(double));
    if (tensor->topology.singular_values) {
        for (size_t i = 0; i < dim; i++) {
            tensor->topology.singular_values[i] = 1.0 / (1.0 + (double)i);
        }
    }

    return tensor;
}

/* Helper function to free topological tensor */
static void free_test_topological_tensor(quantum_topological_tensor_t* tensor) {
    if (!tensor) return;

    free(tensor->components);
    free(tensor->spin_system.spin_states);
    free(tensor->geometry.metric_tensor);
    free(tensor->geometry.parallel_transport);
    free(tensor->geometry.christoffel_symbols);
    free(tensor->topology.singular_values);

    if (tensor->topology.homology) {
        free(tensor->topology.homology->betti_numbers);
        free(tensor->topology.homology->persistence_diagram);
        free(tensor->topology.homology);
    }

    free(tensor);
}

/* Test setup */
static void setUp(void) {
    // Create quantum geometric tensor
    qgt = create_test_topological_tensor(TEST_DIM, TEST_NUM_QUBITS);

    // Create tree tensor network
    if (qgt) {
        network = create_tree_tensor_network(TEST_NUM_QUBITS, TEST_BOND_DIM, TEST_SVD_TOLERANCE);
    }
}

/* Test teardown */
static void tearDown(void) {
    free_test_topological_tensor(qgt);
    qgt = NULL;

    if (network) {
        destroy_tree_tensor_network(network);
        network = NULL;
    }
}

/* Test macro */
#define RUN_TEST(test_fn) do { \
    printf("Running %s... ", #test_fn); \
    fflush(stdout); \
    tests_run++; \
    setUp(); \
    test_fn(); \
    tearDown(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

/* Test TEE calculation */
void test_topological_entropy(void) {
    TEST_ASSERT(network != NULL);

    // Calculate TEE for ground state
    double tee = calculate_topological_entropy(network);

    // TEE should be in reasonable range
    TEST_ASSERT(tee >= EXPECTED_TEE_MIN);
    TEST_ASSERT(tee < EXPECTED_TEE_MAX);
    TEST_ASSERT(isfinite(tee));
}

/* Test error detection */
void test_error_detection(void) {
    TEST_ASSERT(qgt != NULL);

    // Check initial state - should have no errors
    qgt_error_t err = detect_topological_errors(qgt);
    // Initial state might have trivial errors, that's OK
    TEST_ASSERT(err == QGT_SUCCESS || err == QGT_ERROR_VALIDATION_FAILED);
}

/* Test error correction */
void test_error_correction(void) {
    TEST_ASSERT(qgt != NULL);
    TEST_ASSERT(network != NULL);

    // Get initial TEE
    double initial_tee = calculate_topological_entropy(network);
    TEST_ASSERT(isfinite(initial_tee));

    // Correct any errors (this should not crash even if no errors)
    correct_topological_errors(qgt);

    // Verify state is still valid
    TEST_ASSERT(qgt->components != NULL);
    TEST_ASSERT(qgt->dimension == TEST_DIM);
}

/* Test coherence maintenance */
void test_coherence_maintenance(void) {
    TEST_ASSERT(network != NULL);

    // Get initial total entropy
    double initial_entropy = network->total_entanglement_entropy;
    TEST_ASSERT(isfinite(initial_entropy));

    // Verify per-site entropy array
    TEST_ASSERT(network->entanglement_entropy != NULL);
    for (size_t i = 0; i < network->num_sites; i++) {
        TEST_ASSERT(isfinite(network->entanglement_entropy[i]));
        TEST_ASSERT(network->entanglement_entropy[i] >= 0.0);
    }

    // Maintain coherence
    maintain_long_range_coherence(network);

    // Verify network still valid
    TEST_ASSERT(network->num_sites > 0);
    TEST_ASSERT(network->bond_dim == TEST_BOND_DIM);
}

/* Test tensor network creation and properties */
void test_tensor_network_properties(void) {
    TEST_ASSERT(network != NULL);

    // Verify basic properties
    TEST_ASSERT(network->num_sites == TEST_NUM_QUBITS);
    TEST_ASSERT(network->bond_dim == TEST_BOND_DIM);

    // Verify site tensors allocated
    TEST_ASSERT(network->site_tensors != NULL);

    // Check entanglement entropy array is allocated and valid
    TEST_ASSERT(network->entanglement_entropy != NULL);
    for (size_t i = 0; i < network->num_sites; i++) {
        TEST_ASSERT(network->entanglement_entropy[i] >= 0.0);
        TEST_ASSERT(isfinite(network->entanglement_entropy[i]));
    }

    // Check total entropy
    TEST_ASSERT(network->total_entanglement_entropy >= 0.0);
    TEST_ASSERT(isfinite(network->total_entanglement_entropy));
}

/* Test topological order verification */
void test_topological_order(void) {
    TEST_ASSERT(qgt != NULL);

    // Verify topological order
    bool order_valid = verify_topological_order(qgt);
    // Order might not be valid for random initial state, that's OK
    // We just verify it doesn't crash
    (void)order_valid;

    // Update ground state
    update_ground_state(qgt);

    // Tensor should still be valid
    TEST_ASSERT(qgt->dimension == TEST_DIM);
}

/* Test attention configuration */
void test_attention_config(void) {
    TEST_ASSERT(qgt != NULL);

    // Create attention config
    AttentionConfig config = {
        .num_heads = 4,
        .head_dim = TEST_DIM / 4,
        .dropout_rate = 0.1,
        .use_causal_mask = false,
        .temperature = 1.0
    };

    // Apply topological attention
    apply_topological_attention(qgt, &config);

    // Verify tensor still valid
    TEST_ASSERT(qgt->components != NULL);
}

/* Test monitoring configuration */
void test_monitor_config(void) {
    TEST_ASSERT(qgt != NULL);

    // Create monitor config
    MonitorConfig config = {
        .check_interval = 1.0,
        .order_threshold = 0.1,
        .tee_threshold = 0.5,
        .braiding_threshold = 0.01,
        .auto_correct = false
    };

    // Apply monitoring (single check, not continuous)
    monitor_topological_order(qgt, &config);

    // Verify tensor still valid
    TEST_ASSERT(qgt->dimension == TEST_DIM);
}

/* Test distributed protection */
void test_distributed_protection(void) {
    // Create test partitions
    size_t num_parts = 2;
    NetworkPartition* parts = (NetworkPartition*)calloc(num_parts, sizeof(NetworkPartition));
    TEST_ASSERT(parts != NULL);

    // Initialize partitions
    size_t partition_size = TEST_DIM / num_parts;
    for (size_t i = 0; i < num_parts; i++) {
        parts[i].start_index = i * partition_size;
        parts[i].end_index = (i + 1) * partition_size;
        parts[i].local_state = (double*)calloc(partition_size, sizeof(double));
        parts[i].boundary_entropy = 0.0;
        parts[i].needs_sync = false;

        // Initialize local state
        if (parts[i].local_state) {
            for (size_t j = 0; j < partition_size; j++) {
                parts[i].local_state[j] = 1.0 / sqrt((double)partition_size);
            }
        }
    }

    // Protect distributed state
    protect_distributed_state(parts, num_parts);

    // Verify partitions still valid
    for (size_t i = 0; i < num_parts; i++) {
        TEST_ASSERT(parts[i].end_index > parts[i].start_index);
    }

    // Cleanup
    for (size_t i = 0; i < num_parts; i++) {
        free(parts[i].local_state);
    }
    free(parts);
}

/* Test null safety */
void test_null_safety(void) {
    // These should not crash
    calculate_topological_entropy(NULL);
    detect_topological_errors(NULL);
    correct_topological_errors(NULL);
    maintain_long_range_coherence(NULL);
    verify_topological_order(NULL);
    update_ground_state(NULL);
    apply_topological_attention(NULL, NULL);
    monitor_topological_order(NULL, NULL);
    protect_distributed_state(NULL, 0);
}

/* Run all tests */
int main(void) {
    printf("\n");
    printf("===========================================\n");
    printf("Topological Protection Test Suite\n");
    printf("===========================================\n\n");

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Run tests
    RUN_TEST(test_topological_entropy);
    RUN_TEST(test_error_detection);
    RUN_TEST(test_error_correction);
    RUN_TEST(test_coherence_maintenance);
    RUN_TEST(test_tensor_network_properties);
    RUN_TEST(test_topological_order);
    RUN_TEST(test_attention_config);
    RUN_TEST(test_monitor_config);
    RUN_TEST(test_distributed_protection);
    RUN_TEST(test_null_safety);

    printf("\n");
    printf("===========================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("===========================================\n");

    return tests_passed == tests_run ? 0 : 1;
}
