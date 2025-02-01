#include "quantum_geometric_attention.h"
#include "test_helpers.h"
#include <stdio.h>
#include <stdlib.h>

static void test_geometric_attention() {
    printf("Testing geometric attention...\n");
    
    // Setup test data
    quantum_state_t queries, keys, values;
    geometric_tensor_t metric;
    size_t dim = 4;
    
    ASSERT_SUCCESS(allocate_quantum_state(&queries, dim));
    ASSERT_SUCCESS(allocate_quantum_state(&keys, dim));
    ASSERT_SUCCESS(allocate_quantum_state(&values, dim));
    ASSERT_SUCCESS(allocate_geometric_tensor(&metric, dim));
    
    // Initialize with test values
    for (size_t i = 0; i < dim; i++) {
        queries.data[i] = (complex_t){.real = 0.5, .imag = 0.0};
        keys.data[i] = (complex_t){.real = 0.5, .imag = 0.0};
        values.data[i] = (complex_t){.real = 1.0, .imag = 0.0};
    }
    
    // Setup attention inputs
    attention_inputs_t inputs = {
        .queries = &queries,
        .keys = &keys,
        .values = &values,
        .metric = &metric
    };
    
    attention_config_t config = {
        .num_heads = 1,
        .key_dim = dim,
        .value_dim = dim,
        .temperature = 1.0,
        .use_geometric_bias = false
    };
    
    // Test attention computation
    quantum_state_t output;
    ASSERT_SUCCESS(allocate_quantum_state(&output, dim));
    
    ASSERT_SUCCESS(compute_geometric_attention(&inputs, &config, &output));
    
    // Verify output properties
    for (size_t i = 0; i < dim; i++) {
        ASSERT_NEAR(output.data[i].real, 1.0, 1e-6);
        ASSERT_NEAR(output.data[i].imag, 0.0, 1e-6);
    }
    
    // Cleanup
    free_quantum_state(&queries);
    free_quantum_state(&keys);
    free_quantum_state(&values);
    free_quantum_state(&output);
    free_geometric_tensor(&metric);
    
    printf("Geometric attention tests passed!\n");
}

static void test_manifold_attention() {
    printf("Testing manifold attention...\n");
    
    // Setup test manifold
    manifold_t manifold;
    ASSERT_SUCCESS(create_sphere_manifold(2, &manifold));
    
    // Setup test states
    quantum_state_t queries, keys, values;
    size_t dim = 3;
    
    ASSERT_SUCCESS(allocate_quantum_state(&queries, dim));
    ASSERT_SUCCESS(allocate_quantum_state(&keys, dim));
    ASSERT_SUCCESS(allocate_quantum_state(&values, dim));
    
    // Initialize with normalized states on sphere
    queries.data[0] = (complex_t){.real = 1.0/sqrt(2), .imag = 0.0};
    queries.data[1] = (complex_t){.real = 1.0/sqrt(2), .imag = 0.0};
    queries.data[2] = (complex_t){.real = 0.0, .imag = 0.0};
    
    keys.data[0] = (complex_t){.real = 1.0/sqrt(2), .imag = 0.0};
    keys.data[1] = (complex_t){.real = -1.0/sqrt(2), .imag = 0.0};
    keys.data[2] = (complex_t){.real = 0.0, .imag = 0.0};
    
    values.data[0] = (complex_t){.real = 1.0, .imag = 0.0};
    values.data[1] = (complex_t){.real = 0.0, .imag = 0.0};
    values.data[2] = (complex_t){.real = 0.0, .imag = 0.0};
    
    // Setup attention inputs
    attention_inputs_t inputs = {
        .queries = &queries,
        .keys = &keys,
        .values = &values,
        .metric = NULL
    };
    
    attention_config_t config = {
        .num_heads = 1,
        .key_dim = dim,
        .value_dim = dim,
        .temperature = 1.0,
        .use_geometric_bias = true
    };
    
    // Test manifold attention
    quantum_state_t output;
    ASSERT_SUCCESS(allocate_quantum_state(&output, dim));
    
    ASSERT_SUCCESS(compute_manifold_attention(&inputs, &manifold, &config, &output));
    
    // Verify output is on manifold
    double radius;
    ASSERT_SUCCESS(compute_manifold_distance(&output, &manifold, &radius));
    ASSERT_NEAR(radius, 1.0, 1e-6);
    
    // Cleanup
    free_quantum_state(&queries);
    free_quantum_state(&keys);
    free_quantum_state(&values);
    free_quantum_state(&output);
    free_manifold(&manifold);
    
    printf("Manifold attention tests passed!\n");
}

int main() {
    printf("Running quantum geometric attention tests...\n");
    
    test_geometric_attention();
    test_manifold_attention();
    
    printf("All quantum geometric attention tests passed!\n");
    return 0;
}
