/**
 * @file test_language_model.c
 * @brief Tests for language model quantum geometric functionality
 */

#include <quantum_geometric_core.h>
#include <quantum_geometric_tensor_network.h>
#include <quantum_ai_operations.h>
#include <unity.h>
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
static quantum_geometric_tensor* embeddings;

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
    
    // Create embeddings
    embeddings = create_quantum_tensor(
        config.vocab_size,
        config.hidden_dim,
        QGT_MEM_HUGE_PAGES
    );
    TEST_ASSERT_NOT_NULL(embeddings);
    
    // Initialize embeddings
    initialize_geometric_embeddings(embeddings, QGT_EMBED_HYPERBOLIC);
    
    // Create layers
    layers = malloc(config.num_layers * sizeof(TreeTensorNetwork*));
    TEST_ASSERT_NOT_NULL(layers);
    
    for (size_t i = 0; i < config.num_layers; i++) {
        layers[i] = create_transformer_layer(&config);
        TEST_ASSERT_NOT_NULL(layers[i]);
    }
}

void tearDown(void) {
    // Clean up
    free_quantum_tensor(embeddings);
    for (size_t i = 0; i < config.num_layers; i++) {
        physicsml_ttn_destroy(layers[i]);
    }
    free(layers);
}

/* Test geometric properties */
void test_geometric_properties(void) {
    // Create test input
    quantum_geometric_tensor* input = create_quantum_tensor(
        TEST_BATCH_SIZE,
        config.hidden_dim,
        QGT_MEM_HUGE_PAGES
    );
    
    // Initialize with random values
    initialize_random_state(input, 42);
    
    // Forward pass
    TreeTensorNetwork* output = forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );
    TEST_ASSERT_NOT_NULL(output);
    
    // Check geometric properties
    double curvature = calculate_geometric_curvature(output);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, -1.0, curvature);
    
    // Check metric preservation
    quantum_geometric_tensor* final = extract_geometric_properties(output);
    TEST_ASSERT_NOT_NULL(final);
    
    double metric_diff = compare_metric_tensors(input, final);
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, 0.0, metric_diff);
    
    // Clean up
    free_quantum_tensor(input);
    free_quantum_tensor(final);
    physicsml_ttn_destroy(output);
}

/* Test physical constraints */
void test_physical_constraints(void) {
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
        quantum_geometric_tensor* state = extract_geometric_properties(layers[i]);
        TEST_ASSERT_NOT_NULL(state);
        
        qgt_error_t err = apply_physical_constraints(state, &constraints);
        TEST_ASSERT_EQUAL(QGT_SUCCESS, err);
        
        // Verify constraints
        double energy = calculate_total_energy(state);
        TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, constraints.energy_threshold, energy);
        
        bool symmetric = verify_symmetry_constraints(state, constraints.symmetry_tolerance);
        TEST_ASSERT_TRUE(symmetric);
        
        bool causal = verify_causality_constraints(state, constraints.causality_tolerance);
        TEST_ASSERT_TRUE(causal);
        
        free_quantum_tensor(state);
    }
}

/* Test tensor network compression */
void test_tensor_compression(void) {
    // Get original parameter count
    size_t original_params = config.hidden_dim * config.hidden_dim * config.num_layers;
    
    // Get compressed parameter count
    size_t compressed_params = 0;
    for (size_t i = 0; i < config.num_layers; i++) {
        compressed_params += count_network_parameters(layers[i]);
    }
    
    // Verify compression ratio
    double compression_ratio = (double)original_params / compressed_params;
    TEST_ASSERT_FLOAT_WITHIN(1.0, 10.0, compression_ratio);
    
    // Verify quality preservation
    quantum_geometric_tensor* input = create_quantum_tensor(
        TEST_BATCH_SIZE,
        config.hidden_dim,
        QGT_MEM_HUGE_PAGES
    );
    initialize_random_state(input, 42);
    
    // Compare original vs compressed output
    TreeTensorNetwork* compressed_output = forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );
    
    quantum_geometric_tensor* original_output = forward_uncompressed_network(
        layers,
        config.num_layers,
        input
    );
    
    double output_diff = compare_tensor_outputs(
        compressed_output,
        original_output,
        TEST_TOLERANCE
    );
    TEST_ASSERT_FLOAT_WITHIN(TEST_TOLERANCE, 0.0, output_diff);
    
    // Clean up
    free_quantum_tensor(input);
    free_quantum_tensor(original_output);
    physicsml_ttn_destroy(compressed_output);
}

/* Test distributed operations */
void test_distributed_operations(void) {
    // Set up distributed config
    DistributedConfig dist_config = {
        .world_size = 2,
        .pipeline_stages = 2,
        .tensor_parallel = 1,
        .activation_checkpointing = true,
        .zero_optimization_stage = 3,
        .mixed_precision = true
    };
    
    initialize_distributed_training(&dist_config);
    
    // Create distributed input
    quantum_geometric_tensor* input = create_quantum_tensor(
        TEST_BATCH_SIZE,
        config.hidden_dim,
        QGT_MEM_HUGE_PAGES
    );
    initialize_random_state(input, 42);
    
    // Forward pass
    TreeTensorNetwork* output = forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_DISTRIBUTED
    );
    TEST_ASSERT_NOT_NULL(output);
    
    // Verify output consistency
    bool consistent = verify_distributed_consistency(output, TEST_TOLERANCE);
    TEST_ASSERT_TRUE(consistent);
    
    // Clean up
    free_quantum_tensor(input);
    physicsml_ttn_destroy(output);
}

/* Test optimization */
void test_optimization(void) {
    // Create optimizer
    TrainingConfig train_config = {
        .batch_size = TEST_BATCH_SIZE,
        .learning_rate = 1e-4,
        .warmup_steps = 100,
        .total_steps = 1000,
        .weight_decay = 0.1,
        .gradient_clipping = 1.0
    };
    
    GeometricOptimizer* optimizer = create_geometric_optimizer(
        OPTIMIZER_ADAM,
        &train_config,
        QGT_OPT_PRESERVE_GEOMETRY
    );
    TEST_ASSERT_NOT_NULL(optimizer);
    
    // Training step
    quantum_geometric_tensor* input = create_quantum_tensor(
        TEST_BATCH_SIZE,
        config.hidden_dim,
        QGT_MEM_HUGE_PAGES
    );
    initialize_random_state(input, 42);
    
    TreeTensorNetwork* output = forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );
    
    double initial_loss = calculate_geometric_loss(
        output,
        input,
        QGT_LOSS_HYPERBOLIC
    );
    
    backward_geometric_network(
        output,
        initial_loss,
        optimizer,
        QGT_BACKWARD_STANDARD
    );
    
    update_geometric_parameters(
        layers,
        config.num_layers,
        optimizer,
        QGT_UPDATE_PRESERVE_GEOMETRY
    );
    
    // Verify loss improvement
    TreeTensorNetwork* new_output = forward_geometric_network(
        layers,
        config.num_layers,
        input,
        QGT_FORWARD_STANDARD
    );
    
    double final_loss = calculate_geometric_loss(
        new_output,
        input,
        QGT_LOSS_HYPERBOLIC
    );
    
    TEST_ASSERT_TRUE(final_loss < initial_loss);
    
    // Clean up
    free_quantum_tensor(input);
    physicsml_ttn_destroy(output);
    physicsml_ttn_destroy(new_output);
    free_geometric_optimizer(optimizer);
}

/* Run all tests */
int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_geometric_properties);
    RUN_TEST(test_physical_constraints);
    RUN_TEST(test_tensor_compression);
    RUN_TEST(test_distributed_operations);
    RUN_TEST(test_optimization);
    
    return UNITY_END();
}
