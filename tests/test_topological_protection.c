/**
 * @file test_topological_protection.c
 * @brief Tests for topological error correction and coherence protection
 */

#include "../include/topological_protection.h"
#include <unity.h>
#include <math.h>

/* Test configuration */
#define TEST_DIM 32
#define TEST_BOND_DIM 8
#define TEST_NUM_QUBITS 16
#define TEST_TOLERANCE 1e-6

/* Test fixtures */
static quantum_geometric_tensor* qgt;
static TreeTensorNetwork* network;

void setUp(void) {
    // Create quantum geometric tensor
    qgt = create_quantum_tensor(TEST_DIM, TEST_NUM_QUBITS, QGT_MEM_HUGE_PAGES);
    TEST_ASSERT_NOT_NULL(qgt);
    
    // Initialize with test state
    initialize_test_state(qgt);
    
    // Create tensor network
    network = create_geometric_network(qgt, TEST_BOND_DIM);
    TEST_ASSERT_NOT_NULL(network);
}

void tearDown(void) {
    free_quantum_tensor(qgt);
    physicsml_ttn_destroy(network);
}

/* Test TEE calculation */
void test_topological_entropy(void) {
    // Calculate TEE for ground state
    double tee = calculate_topological_entropy(network);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, EXPECTED_TEE, tee);
    
    // Introduce error
    introduce_test_error(qgt);
    
    // Verify TEE changes
    double tee_error = calculate_topological_entropy(network);
    TEST_ASSERT_TRUE(fabs(tee_error - EXPECTED_TEE) > TEST_TOLERANCE);
}

/* Test error detection */
void test_error_detection(void) {
    // Check initial state
    ErrorCode err = detect_topological_errors(qgt);
    TEST_ASSERT_EQUAL(NO_ERROR, err);
    
    // Introduce errors
    introduce_test_error(qgt);
    
    // Verify error detection
    err = detect_topological_errors(qgt);
    TEST_ASSERT_EQUAL(ERROR_DETECTED, err);
}

/* Test error correction */
void test_error_correction(void) {
    // Get initial TEE
    double initial_tee = calculate_topological_entropy(network);
    
    // Introduce error
    introduce_test_error(qgt);
    
    // Verify error detected
    double error_tee = calculate_topological_entropy(network);
    TEST_ASSERT_TRUE(fabs(error_tee - initial_tee) > TEST_TOLERANCE);
    
    // Correct error
    correct_topological_errors(qgt);
    
    // Verify correction
    double final_tee = calculate_topological_entropy(network);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, initial_tee, final_tee);
}

/* Test anyon braiding */
void test_anyon_braiding(void) {
    // Create test anyons
    AnyonExcitation* anyons = create_test_anyons(qgt);
    TEST_ASSERT_NOT_NULL(anyons);
    
    // Calculate initial braiding statistics
    double initial_stats = verify_braiding_statistics(qgt);
    
    // Apply braiding
    BraidingPattern* pattern = calculate_braiding_pattern(anyons);
    TEST_ASSERT_NOT_NULL(pattern);
    
    apply_braiding_correction(qgt, pattern);
    
    // Verify statistics preserved
    double final_stats = verify_braiding_statistics(qgt);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, initial_stats, final_stats);
    
    // Clean up
    free_braiding_pattern(pattern);
    free_anyon_excitations(anyons);
}

/* Test coherence maintenance */
void test_coherence_maintenance(void) {
    // Get initial correlation length
    double xi_initial = calculate_correlation_length(network);
    
    // Evolve system
    evolve_test_system(network);
    
    // Check correlation length increased
    double xi_evolved = calculate_correlation_length(network);
    TEST_ASSERT_TRUE(xi_evolved > xi_initial);
    
    // Maintain coherence
    maintain_long_range_coherence(network);
    
    // Verify correlation length restored
    double xi_final = calculate_correlation_length(network);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, xi_initial, xi_final);
}

/* Test distributed protection */
void test_distributed_protection(void) {
    // Create partitions
    size_t num_parts = 4;
    NetworkPartition* parts = create_test_partitions(network, num_parts);
    TEST_ASSERT_NOT_NULL(parts);
    
    // Calculate initial global TEE
    double initial_tee = 0.0;
    for (size_t i = 0; i < num_parts; i++) {
        initial_tee += calculate_partition_tee(&parts[i]);
    }
    
    // Introduce distributed errors
    introduce_distributed_errors(parts, num_parts);
    
    // Protect state
    protect_distributed_state(parts, num_parts);
    
    // Verify protection
    double final_tee = 0.0;
    for (size_t i = 0; i < num_parts; i++) {
        final_tee += calculate_partition_tee(&parts[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, initial_tee, final_tee);
    
    // Clean up
    free_network_partitions(parts, num_parts);
}

/* Test topological attention */
void test_topological_attention(void) {
    // Create attention config
    AttentionConfig config = {
        .num_heads = 8,
        .head_dim = TEST_DIM / 8,
        .attention_scale = 1.0 / sqrt(TEST_DIM / 8),
        .preserve_tee = true
    };
    
    // Get initial TEE
    double initial_tee = calculate_topological_entropy(network);
    
    // Apply attention
    apply_topological_attention(qgt, &config);
    
    // Verify TEE preserved
    double final_tee = calculate_topological_entropy(network);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, initial_tee, final_tee);
}

/* Test monitoring */
void test_monitoring(void) {
    // Create monitor config
    MonitorConfig config = {
        .check_interval = 0.1,
        .correction_threshold = 0.1,
        .active = true,
        .needs_correction = false
    };
    
    // Start monitoring in separate thread
    pthread_t monitor_thread;
    pthread_create(&monitor_thread, NULL, 
                  (void*)monitor_topological_order, 
                  (void*)&config);
    
    // Introduce periodic errors
    for (int i = 0; i < 10; i++) {
        // Add error
        introduce_test_error(qgt);
        
        // Wait for correction
        usleep(200000);  // 200ms
        
        // Verify correction
        ErrorCode err = detect_topological_errors(qgt);
        TEST_ASSERT_EQUAL(NO_ERROR, err);
    }
    
    // Stop monitoring
    config.active = false;
    pthread_join(monitor_thread, NULL);
}

/* Run all tests */
int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_topological_entropy);
    RUN_TEST(test_error_detection);
    RUN_TEST(test_error_correction);
    RUN_TEST(test_anyon_braiding);
    RUN_TEST(test_coherence_maintenance);
    RUN_TEST(test_distributed_protection);
    RUN_TEST(test_topological_attention);
    RUN_TEST(test_monitoring);
    
    return UNITY_END();
}
