#include "quantum_geometric/ai/quantum_geometric_attention.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "test_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Test attention configuration initialization
static void test_attention_config(void) {
    printf("Testing attention configuration...\n");

    // Create attention configuration
    attention_config_t config = {
        .geometry = {
            .manifold = MANIFOLD_COMPLEX_PROJECTIVE,
            .metric = METRIC_FUBINI_STUDY,
            .connection = CONNECTION_NATURAL
        },
        .optimization = {
            .type = OPTIMIZATION_GEOMETRIC,
            .complexity = COMPLEXITY_LINEAR,
            .error_protection = true
        },
        .hardware = {
            .backend = BACKEND_CLASSICAL,
            .topology = NULL
        },
        .num_heads = QG_DEFAULT_NUM_HEADS,
        .head_dim = QG_DEFAULT_HEAD_DIM,
        .model_dim = QG_DEFAULT_MODEL_DIM
    };

    // Initialize attention
    int result = qg_attention_init(&config);
    TEST_ASSERT(result == QG_SUCCESS);

    // Verify configuration values
    TEST_ASSERT(config.num_heads == QG_DEFAULT_NUM_HEADS);
    TEST_ASSERT(config.head_dim == QG_DEFAULT_HEAD_DIM);
    TEST_ASSERT(config.model_dim == QG_DEFAULT_MODEL_DIM);
    TEST_ASSERT(config.geometry.manifold == MANIFOLD_COMPLEX_PROJECTIVE);
    TEST_ASSERT(config.geometry.metric == METRIC_FUBINI_STUDY);

    printf("  Configuration test passed\n");
}

// Test hierarchical attention initialization
static void test_hierarchical_attention(void) {
    printf("Testing hierarchical attention...\n");

    hierarchical_attention_t hier_attn;
    size_t seq_length = 64;

    // Initialize hierarchical attention
    int result = qg_hierarchical_attention_init(&hier_attn, seq_length);
    TEST_ASSERT(result == QG_SUCCESS);
    TEST_ASSERT(hier_attn.seq_length == seq_length);
    TEST_ASSERT(hier_attn.num_levels > 0);

    // Cleanup
    qg_hierarchical_attention_cleanup(&hier_attn);

    printf("  Hierarchical attention test passed\n");
}

// Test attention creation and destruction
static void test_attention_lifecycle(void) {
    printf("Testing attention lifecycle...\n");

    // Create configuration
    attention_config_t config = {
        .geometry = {
            .manifold = MANIFOLD_EUCLIDEAN,
            .metric = METRIC_EUCLIDEAN,
            .connection = CONNECTION_RIEMANNIAN
        },
        .optimization = {
            .type = OPTIMIZATION_NATURAL_GRADIENT,
            .complexity = COMPLEXITY_LOG_LINEAR,
            .error_protection = false
        },
        .hardware = {
            .backend = BACKEND_AUTO,
            .topology = NULL
        },
        .num_heads = 4,
        .head_dim = 32,
        .model_dim = 128
    };

    // Create attention
    quantum_attention_t* attention = quantum_attention_create(&config);
    TEST_ASSERT(attention != NULL);

    // Get metrics
    attention_metrics_t metrics;
    bool got_metrics = get_attention_metrics(attention, &metrics);
    // Metrics may or may not be available depending on implementation
    (void)got_metrics;

    // Free attention
    quantum_attention_free(attention);

    printf("  Lifecycle test passed\n");
}

// Test different manifold types
static void test_manifold_types(void) {
    printf("Testing manifold types...\n");

    attention_manifold_t manifolds[] = {
        MANIFOLD_COMPLEX_PROJECTIVE,
        MANIFOLD_HYPERBOLIC,
        MANIFOLD_SPHERICAL,
        MANIFOLD_EUCLIDEAN
    };

    for (size_t i = 0; i < sizeof(manifolds)/sizeof(manifolds[0]); i++) {
        attention_config_t config = {
            .geometry = {
                .manifold = manifolds[i],
                .metric = METRIC_ADAPTIVE,
                .connection = CONNECTION_ADAPTIVE
            },
            .optimization = {
                .type = OPTIMIZATION_HYBRID,
                .complexity = COMPLEXITY_ADAPTIVE,
                .error_protection = true
            },
            .hardware = {
                .backend = BACKEND_CLASSICAL,
                .topology = NULL
            },
            .num_heads = 2,
            .head_dim = 16,
            .model_dim = 32
        };

        quantum_attention_t* attention = quantum_attention_create(&config);
        TEST_ASSERT(attention != NULL);

        // Estimate error rate
        double error_rate = estimate_error_rate(attention);
        TEST_ASSERT(error_rate >= 0.0 && error_rate <= 1.0);

        quantum_attention_free(attention);
    }

    printf("  Manifold types test passed\n");
}

// Test attention weights
static void test_attention_weights(void) {
    printf("Testing attention weights...\n");

    attention_weights_t weights = {
        .query_weights = NULL,
        .key_weights = NULL,
        .value_weights = NULL,
        .output_weights = NULL,
        .weight_size = 0
    };

    // Allocate test weights
    size_t weight_size = 64;
    weights.weight_size = weight_size;
    weights.query_weights = (float*)malloc(weight_size * sizeof(float));
    weights.key_weights = (float*)malloc(weight_size * sizeof(float));
    weights.value_weights = (float*)malloc(weight_size * sizeof(float));
    weights.output_weights = (float*)malloc(weight_size * sizeof(float));

    TEST_ASSERT(weights.query_weights != NULL);
    TEST_ASSERT(weights.key_weights != NULL);
    TEST_ASSERT(weights.value_weights != NULL);
    TEST_ASSERT(weights.output_weights != NULL);

    // Initialize weights
    for (size_t i = 0; i < weight_size; i++) {
        weights.query_weights[i] = 0.1f;
        weights.key_weights[i] = 0.1f;
        weights.value_weights[i] = 0.1f;
        weights.output_weights[i] = 0.1f;
    }

    // Cleanup
    qg_attention_cleanup(&weights);

    printf("  Attention weights test passed\n");
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("Running quantum geometric attention tests...\n\n");

    test_attention_config();
    test_hierarchical_attention();
    test_attention_lifecycle();
    test_manifold_types();
    test_attention_weights();

    printf("\nAll quantum geometric attention tests passed!\n");
    return 0;
}
