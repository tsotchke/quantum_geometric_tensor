/**
 * @file test_language_model.c
 * @brief Tests for language model quantum geometric functionality
 */

#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_tensor_network.h"
#include "quantum_geometric/ai/quantum_ai_operations.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Test configuration */
#define TEST_HIDDEN_DIM 768
#define TEST_NUM_LAYERS 4
#define TEST_NUM_HEADS 12
#define TEST_VOCAB_SIZE 1000
#define TEST_SEQ_LENGTH 128
#define TEST_BOND_DIM 32
#define TEST_BATCH_SIZE 8
#define TEST_TOLERANCE 1e-5

/* Test fixtures */
static ModelConfig config;
static TreeTensorNetwork** layers;
static qgt_advanced_geometry_t* embeddings;

void setUp(void) {
    // Initialize test configuration
    config = (ModelConfig) {
        .hidden_dim = TEST_HIDDEN_DIM,
        .num_layers = TEST_NUM_LAYERS,
        .num_heads = TEST_NUM_HEADS,
        .vocab_size = TEST_VOCAB_SIZE,
        .seq_length = TEST_SEQ_LENGTH,
        .bond_dim = TEST_BOND_DIM
    };

    // Create embeddings (dimension first, num_spins second)
    embeddings = qgt_ai_create_tensor(
        config.hidden_dim,
        config.vocab_size,
        QGT_MEM_HUGE_PAGES
    );
    assert(embeddings != NULL);

    // Initialize embeddings
    qgt_ai_initialize_geometric_embeddings(embeddings, QGT_EMBED_HYPERBOLIC);

    // Create layers
    layers = malloc(config.num_layers * sizeof(TreeTensorNetwork*));
    assert(layers != NULL);

    for (size_t i = 0; i < config.num_layers; i++) {
        layers[i] = qgt_ai_create_transformer_layer(&config);
        assert(layers[i] != NULL);
    }
}

void tearDown(void) {
    // Clean up
    qgt_ai_free_tensor(embeddings);
    for (size_t i = 0; i < config.num_layers; i++) {
        qgt_ai_destroy_ttn(layers[i]);
    }
    free(layers);
}

/* Test geometric properties */
static void test_geometric_properties(void) {
    printf("Testing geometric properties...\n");

    // Create test input
    qgt_advanced_geometry_t* input = qgt_ai_create_tensor(
        config.hidden_dim,
        TEST_BATCH_SIZE,
        QGT_MEM_HUGE_PAGES
    );

    // Initialize with random values
    qgt_ai_initialize_random_state(input, 42);

    // Forward pass
    TreeTensorNetwork* output = qgt_ai_forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );
    assert(output != NULL);

    // Check geometric properties
    double curvature = qgt_ai_calculate_geometric_curvature(output);
    assert(fabs(curvature - (-1.0)) <= TEST_TOLERANCE);

    // Check metric preservation
    qgt_advanced_geometry_t* final = qgt_ai_extract_geometric_properties(output);
    assert(final != NULL);

    double metric_diff = qgt_ai_compare_metric_tensors(input, final);
    assert(fabs(metric_diff - 0.0) <= TEST_TOLERANCE);

    // Clean up
    qgt_ai_free_tensor(input);
    qgt_ai_free_tensor(final);
    qgt_ai_destroy_ttn(output);

    printf("Geometric properties tests passed\n");
}

/* Test physical constraints */
static void test_physical_constraints(void) {
    printf("Testing physical constraints...\n");

    // Create constraints
    PhysicalConstraints constraints = {
        .energy_threshold = 1.0,
        .symmetry_tolerance = 1e-6,
        .conservation_tolerance = 1e-6,
        .gauge_tolerance = 1e-6,
        .locality_tolerance = 1e-6,
        .renormalization_scale = 1.0,
        .causality_tolerance = 1e-6
    };

    // Apply constraints to each layer
    for (size_t i = 0; i < config.num_layers; i++) {
        qgt_advanced_geometry_t* state = qgt_ai_extract_geometric_properties(layers[i]);
        assert(state != NULL);

        qgt_error_t err = qgt_ai_apply_physical_constraints(state, &constraints);
        assert(err == QGT_SUCCESS);

        // Verify constraints
        double energy = qgt_ai_calculate_total_energy(state);
        assert(fabs(energy - constraints.energy_threshold) <= TEST_TOLERANCE);

        bool symmetric = qgt_ai_verify_symmetry_constraints(state, constraints.symmetry_tolerance);
        assert(symmetric);

        bool causal = qgt_ai_verify_causality_constraints(state, constraints.causality_tolerance);
        assert(causal);

        qgt_ai_free_tensor(state);
    }

    printf("Physical constraints tests passed\n");
}

/* Test tensor network compression */
static void test_tensor_compression(void) {
    printf("Testing tensor network compression...\n");

    // Get original parameter count
    size_t original_params = config.hidden_dim * config.hidden_dim * config.num_layers;

    // Get compressed parameter count
    size_t compressed_params = 0;
    for (size_t i = 0; i < config.num_layers; i++) {
        compressed_params += qgt_ai_count_network_parameters(layers[i]);
    }

    // Verify compression ratio
    double compression_ratio = (double)original_params / compressed_params;
    assert(fabs(compression_ratio - 10.0) <= 1.0);

    // Verify quality preservation
    qgt_advanced_geometry_t* input = qgt_ai_create_tensor(
        config.hidden_dim,
        TEST_BATCH_SIZE,
        QGT_MEM_HUGE_PAGES
    );
    qgt_ai_initialize_random_state(input, 42);

    // Compare original vs compressed output
    TreeTensorNetwork* compressed_output = qgt_ai_forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );

    TreeTensorNetwork* uncompressed_output = qgt_ai_forward_uncompressed_network(
        layers,
        config.num_layers,
        input
    );

    qgt_advanced_geometry_t* uncompressed_tensor = qgt_ai_extract_geometric_properties(uncompressed_output);

    double output_diff = qgt_ai_compare_tensor_outputs(
        compressed_output,
        uncompressed_tensor,
        TEST_TOLERANCE
    );
    assert(fabs(output_diff - 0.0) <= TEST_TOLERANCE);

    // Clean up
    qgt_ai_free_tensor(input);
    qgt_ai_free_tensor(uncompressed_tensor);
    qgt_ai_destroy_ttn(compressed_output);
    qgt_ai_destroy_ttn(uncompressed_output);

    printf("Tensor network compression tests passed\n");
}

/* Test distributed operations */
static void test_distributed_operations(void) {
    printf("Testing distributed operations...\n");

    // Set up distributed config
    DistributedConfig dist_config = {
        .world_size = 2,
        .pipeline_stages = 2,
        .tensor_parallel = 1,
        .activation_checkpointing = true,
        .zero_optimization_stage = 3,
        .mixed_precision = true
    };

    qgt_ai_initialize_distributed_training(&dist_config);

    // Create distributed input
    qgt_advanced_geometry_t* input = qgt_ai_create_tensor(
        config.hidden_dim,
        TEST_BATCH_SIZE,
        QGT_MEM_HUGE_PAGES
    );
    qgt_ai_initialize_random_state(input, 42);

    // Forward pass
    TreeTensorNetwork* output = qgt_ai_forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_DISTRIBUTED
    );
    assert(output != NULL);

    // Verify output consistency
    bool consistent = qgt_ai_verify_distributed_consistency(output, TEST_TOLERANCE);
    assert(consistent);

    // Clean up
    qgt_ai_free_tensor(input);
    qgt_ai_destroy_ttn(output);

    printf("Distributed operations tests passed\n");
}

/* Test optimization */
static void test_optimization(void) {
    printf("Testing optimization...\n");

    // Create optimizer
    TrainingConfig train_config = {
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 1e-4,
        .warmup_steps = 100,
        .total_steps = 1000,
        .weight_decay = 0.1,
        .gradient_clipping = 1.0
    };

    GeometricOptimizer* optimizer = qgt_ai_create_geometric_optimizer(
        OPTIMIZER_ADAM,
        &train_config,
        QGT_UPDATE_PRESERVE_GEOMETRY
    );
    assert(optimizer != NULL);

    // Training step
    qgt_advanced_geometry_t* input = qgt_ai_create_tensor(
        config.hidden_dim,
        TEST_BATCH_SIZE,
        QGT_MEM_HUGE_PAGES
    );
    qgt_ai_initialize_random_state(input, 42);

    TreeTensorNetwork* output = qgt_ai_forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );

    double initial_loss = qgt_ai_calculate_geometric_loss(
        output,
        input,
        QGT_LOSS_HYPERBOLIC
    );

    qgt_ai_backward_geometric_network(
        output,
        initial_loss,
        optimizer,
        QGT_BACKWARD_STANDARD
    );

    qgt_ai_update_geometric_parameters(
        layers,
        config.num_layers,
        optimizer,
        QGT_UPDATE_PRESERVE_GEOMETRY
    );

    // Verify loss improvement
    TreeTensorNetwork* new_output = qgt_ai_forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );

    double final_loss = qgt_ai_calculate_geometric_loss(
        new_output,
        input,
        QGT_LOSS_HYPERBOLIC
    );

    assert(final_loss < initial_loss);

    // Clean up
    qgt_ai_free_tensor(input);
    qgt_ai_destroy_ttn(output);
    qgt_ai_destroy_ttn(new_output);
    qgt_ai_free_geometric_optimizer(optimizer);

    printf("Optimization tests passed\n");
}

/* Run all tests */
int main(void) {
    printf("Running language model tests...\n\n");

    setUp();

    test_geometric_properties();
    test_physical_constraints();
    test_tensor_compression();
    test_distributed_operations();
    test_optimization();

    tearDown();

    printf("\nAll language model tests passed!\n");
    return 0;
}
